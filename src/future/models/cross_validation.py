import numpy as np
from math import ceil
from models import utils
from config import cfg
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold


class CrossValidation(object):
    """
        Cross Validation
    """
    def __init__(self):

        self.trained_cv = []

    @staticmethod
    def random_split(x, y, w, e, n_valid=None, n_cv=None, n_era=None, cv_seed=None):

        test_size = n_valid / n_era
        valid_era = []
        ss_train = StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size, random_state=cv_seed)
        cv_count = 0

        for train_index, valid_index in ss_train.split(x, y):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]
            e_train = e[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]
            e_valid = e[valid_index]

            cv_count += 1
            utils.print_cv_info(cv_count, n_cv)

            yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    @staticmethod
    def sk_k_fold(x, y, w, n_splits=None, n_cv=None, cv_seed=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        if n_cv % n_splits != 0:
            raise ValueError('n_cv must be an integer multiple of n_splits!')

        n_repeats = int(n_cv / n_splits)
        era_k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed)
        cv_count = 0

        for train_index, valid_index in era_k_fold.split(x, y):

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            cv_count += 1
            utils.print_cv_info(cv_count, n_cv)

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid

    @staticmethod
    def sk_group_k_fold(x, y, e, n_cv=None):

        era_k_fold = GroupKFold(n_splits=n_cv)
        cv_count = 0

        for train_index, valid_index in era_k_fold.split(x, y, e):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]

            cv_count += 1
            utils.print_cv_info(cv_count, n_cv)

            yield x_train, y_train, x_valid, y_valid

    @staticmethod
    def sk_group_k_fold_with_weight(x, y, w, e, n_cv=None):

        era_k_fold = GroupKFold(n_splits=n_cv)
        cv_count = 0

        for train_index, valid_index in era_k_fold.split(x, y, e):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            cv_count += 1
            utils.print_cv_info(cv_count, n_cv)

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid

    @staticmethod
    def era_k_fold(x, y, w, e, n_valid=None, n_cv=None, n_era=None, cv_seed=None, era_list=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse
        trained_cv = []
        cv_count = 0

        for epoch in range(n_epoch):

            if era_list is None:
                era_list = range(0, n_era)

            era_idx = [era_list]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(era_list, n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):
                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    # Training data
                    x_train = x[train_index]
                    y_train = y[train_index]
                    w_train = w[train_index]
                    e_train = e[train_index]

                    # Validation data
                    x_valid = x[valid_index]
                    y_valid = y[valid_index]
                    w_valid = w[valid_index]
                    e_valid = e[valid_index]

                    trained_cv.append(set(valid_era))

                    cv_count += 1
                    utils.print_cv_info(cv_count, n_cv)

                    yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]

                        trained_cv.append(set(valid_era))

                        cv_count += 1
                        utils.print_cv_info(cv_count, n_cv)

                        yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

                    else:

                        era_idx_else = [t for t in list(era_list) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]

                        trained_cv.append(set(valid_era))

                        cv_count += 1
                        utils.print_cv_info(cv_count, n_cv)

                        yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    @staticmethod
    def era_k_fold_split(e, n_valid=None, n_cv=None, n_era=None, cv_seed=None, era_list=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse
        trained_cv = []
        cv_count = 0

        for epoch in range(n_epoch):

            if era_list is None:
                era_list = range(0, n_era)

            era_idx = [era_list]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(era_list, n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):
                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    trained_cv.append(set(valid_era))

                    cv_count += 1
                    utils.print_cv_info(cv_count, n_cv)

                    yield train_index, valid_index

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        trained_cv.append(set(valid_era))

                        cv_count += 1
                        utils.print_cv_info(cv_count, n_cv)

                        yield train_index, valid_index

                    else:

                        era_idx_else = [t for t in list(range(0, n_era)) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        trained_cv.append(set(valid_era))

                        cv_count += 1
                        utils.print_cv_info(cv_count, n_cv)

                        yield train_index, valid_index

    def era_k_fold_for_stack(self, x, y, w, e, x_g, n_valid=None, n_cv=None,
                             n_era=None, cv_seed=None, return_train_index=False):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse
        cv_count = 0

        for epoch in range(n_epoch):

            era_idx = [list(range(0, n_era))]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if self.trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(range(0, n_era), n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):
                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    # Training data
                    x_train = x[train_index]
                    y_train = y[train_index]
                    w_train = w[train_index]
                    e_train = e[train_index]
                    x_g_train = x_g[train_index]

                    # Validation data
                    x_valid = x[valid_index]
                    y_valid = y[valid_index]
                    w_valid = w[valid_index]
                    e_valid = e[valid_index]
                    x_g_valid = x_g[valid_index]

                    self.trained_cv.append(set(valid_era))

                    cv_count += 1
                    utils.print_cv_info(cv_count, n_cv)

                    if return_train_index:
                        yield x_train, y_train, w_train, e_train, x_g_train, x_valid, \
                              y_valid, w_valid, e_valid, x_g_valid, train_index, valid_index, valid_era
                    else:
                        yield x_train, y_train, w_train, x_g_train, x_valid, \
                              y_valid, w_valid, x_g_valid, valid_index, valid_era

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if self.trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]
                        x_g_train = x_g[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]
                        x_g_valid = x_g[valid_index]

                        self.trained_cv.append(set(valid_era))

                        cv_count += 1
                        utils.print_cv_info(cv_count, n_cv)

                        if return_train_index:
                            yield x_train, y_train, w_train, e_train, x_g_train, x_valid, \
                                  y_valid, w_valid, e_valid, x_g_valid, train_index, valid_index, valid_era
                        else:
                            yield x_train, y_train, w_train, x_g_train, x_valid, \
                                  y_valid, w_valid, x_g_valid, valid_index, valid_era

                    else:

                        era_idx_else = [t for t in list(range(0, n_era)) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(
                            np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):
                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]
                        x_g_train = x_g[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]
                        x_g_valid = x_g[valid_index]

                        self.trained_cv.append(set(valid_era))

                        cv_count += 1
                        utils.print_cv_info(cv_count, n_cv)

                        if return_train_index:
                            yield x_train, y_train, w_train, e_train, x_g_train, x_valid, \
                                  y_valid, w_valid, e_valid, x_g_valid, train_index, valid_index, valid_era
                        else:
                            yield x_train, y_train, w_train, x_g_train, x_valid, \
                                  y_valid, w_valid, x_g_valid, valid_index, valid_era

    @staticmethod
    def era_k_fold_balance(x, y, w, e, n_valid=None, n_cv=None, n_era=None, cv_seed=None, era_list=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        trained_cv = []

        for i in range(n_cv):

            if era_list is None:
                era_list = range(0, n_era)

            valid_era = np.random.choice(era_list, n_valid, replace=False)
            while utils.check_bad_cv(trained_cv, valid_era):
                valid_era = np.random.choice(era_list, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):
                if ele in valid_era:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]
            e_train = e[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]
            e_valid = e[valid_index]

            trained_cv.append(set(valid_era))

            utils.print_cv_info(i+1, n_cv)

            yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    @staticmethod
    def era_k_fold_all_random(x, y, w, e, n_valid=None, n_cv=None, n_era=None, cv_seed=None, era_list=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        trained_cv = []

        for i in range(n_cv):

            if era_list is None:
                era_list = range(0, n_era)

            valid_era = np.random.choice(era_list, n_valid, replace=False)
            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                print('This CV split has been chosen, choosing another one...')
                valid_era = np.random.choice(era_list, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):
                if ele in valid_era:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]
            e_train = e[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]
            e_valid = e[valid_index]

            trained_cv.append(set(valid_era))

            utils.print_cv_info(i+1, n_cv)

            yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    @staticmethod
    def era_k_fold_split_all_random(e, n_valid=None, n_cv=None, n_era=None, cv_seed=None, era_list=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        trained_cv = []

        for i in range(n_cv):

            if era_list is None:
                era_list = range(0, n_era)

            valid_era = np.random.choice(era_list, n_valid, replace=False)
            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                print('This CV split has been chosen, choosing another one...')
                valid_era = np.random.choice(era_list, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):

                if ele in valid_era:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            trained_cv.append(set(valid_era))

            utils.print_cv_info(i+1, n_cv)

            yield train_index, valid_index

    @staticmethod
    def forward_increase(x, y, w, e, n_valid=None, n_cv=None, n_era=None, cv_seed=None, valid_rate=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        # If valid_rate is provided, dynamically calculate n_valid
        if valid_rate is not None:
            n_valid_last = ceil(n_era*valid_rate)
        else:
            n_valid_last = n_valid
        step = (n_era-n_valid_last)//n_cv

        for i in range(n_cv):

            valid_start = (i+1) * step
            # If valid_rate is provided, dynamically calculate n_valid
            if valid_rate is not None:
                n_valid = ceil((valid_start*valid_rate)/(1-valid_rate))

            if i == (n_cv - 1):
                valid_stop = n_era
            else:
                valid_stop = valid_start + n_valid

            print('======================================================')
            print('Train Era: {}-{}'.format(0, valid_start-1))
            print('Valid Era: {}-{}'.format(valid_start, valid_stop-1))

            train_era = range(0, valid_start)
            valid_era = list(range(valid_start, valid_stop))

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):
                if ele in train_era:
                    train_index.append(ii)
                elif ele in valid_era:
                    valid_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]
            e_train = e[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]
            e_valid = e[valid_index]

            utils.print_cv_info(i+1, n_cv)

            yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    @staticmethod
    def forward_window(x, y, w, e, n_valid=None, n_cv=None, n_era=None,
                       window_size=None, cv_seed=None, valid_rate=None, ensemble=False):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        if ensemble:
            x_valid = utils.load_preprocessed_data(cfg.preprocessed_data_path)[4]
            y_valid, w_valid, e_valid = utils.load_preprocessed_test_data(cfg.preprocessed_data_path)
        else:
            x_valid, y_valid, w_valid, e_valid = None, None, None, None

        n_step = (n_era-window_size) // n_cv
        if valid_rate is not None:
            n_valid = ceil(window_size*valid_rate)
        train_start = 0

        for i in range(n_cv):

            if ensemble:

                if i == (n_cv - 1):
                    train_start = n_era - window_size
                    train_end = n_era
                else:
                    train_end = train_start + window_size

                print('======================================================')
                print('Train Era: {}-{}'.format(train_start, train_end - 1))

                train_era = list(range(train_start, train_end))
                valid_era = list(set(e_valid))
                train_index = []
                for ii, ele in enumerate(e):
                    if ele in train_era:
                        train_index.append(ii)
                np.random.shuffle(train_index)

                # Training data
                x_train = x[train_index]
                y_train = y[train_index]
                w_train = w[train_index]
                e_train = e[train_index]

                train_start += n_step
                utils.print_cv_info(i + 1, n_cv)

                yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

            else:
                if i == (n_cv - 1):
                    train_start = n_era - window_size
                    train_end = n_era - n_valid
                    valid_stop = n_era
                else:
                    train_end = train_start + window_size - n_valid
                    valid_stop = train_start + window_size

                print('======================================================')
                print('Train Era: {}-{}'.format(train_start, train_end - 1))
                print('Valid Era: {}-{}'.format(train_end, valid_stop - 1))

                train_era = list(range(train_start, train_end))
                valid_era = list(range(train_end, valid_stop))

                train_index = []
                valid_index = []

                for ii, ele in enumerate(e):
                    if ele in train_era:
                        train_index.append(ii)
                    elif ele in valid_era:
                        valid_index.append(ii)

                np.random.shuffle(train_index)
                np.random.shuffle(valid_index)

                # Training data
                x_train = x[train_index]
                y_train = y[train_index]
                w_train = w[train_index]
                e_train = e[train_index]

                # Validation data
                x_valid = x[valid_index]
                y_valid = y[valid_index]
                w_valid = w[valid_index]
                e_valid = e[valid_index]

                train_start += n_step
                utils.print_cv_info(i+1, n_cv)

                yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era
