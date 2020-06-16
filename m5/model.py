import lightgbm as lgb


def train(params, train, valid, **fit_params):
    if not isinstance(valid, list):
        valid = [valid]
    valid = [train] + valid
    return lgb.train(
        params,
        train_set=train,
        valid_sets=valid,
        **fit_params
    )
