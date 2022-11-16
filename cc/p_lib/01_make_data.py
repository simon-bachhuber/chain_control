

def make_data(
    env_config: dict, 
    cache_folder = ".cache/replay_samples", 
    val=False, 
    test=False,
    train_gp = [0,1,2,3,4,5,6,7,8,9,10,11],
    train_cos =  [1,2,3,4,5,6,7,8,9,10,12,14],
    val_gp = [15,16,17,18], 
    val_cos = [2.5, 5.0, 7.5, 10.0],
    test_gp = [16,17,18,19],
    test_cos = [3.5, 5.5, 7.5, 9.5]
    ):

    path = Path(cache_folder); path.mkdir(exist_ok=True, parents=True)

    naming = lambda train_val_test: path.joinpath(f"sample_{train_val_test}||{name_from_config(env_config)}.pkl")

    has_train = has_val = has_test = False
    sample_val = sample_test = None 

    try: 
        sample_train = load(naming("train"))
        has_train = True 
        if val:
            sample_val = load(naming("val"))
            has_val = True 
        if test:
            sample_test = load(naming("test"))
            has_test = True 

    except:
        env = make_env(**env_config)
        
        if not has_train:
            sample_train = sample_feedforward_and_collect(env, train_gp, train_cos)
            save(sample_train, naming("train"))

        if val:
            if not has_val:
                sample_val = sample_feedforward_and_collect(env, val_gp, val_cos)
                save(sample_val, naming("val"))

        if test:
            if not has_test:
                sample_test = sample_feedforward_and_collect(env, test_gp, test_cos)
                save(sample_test, naming("test"))

    return sample_train, sample_val, sample_test 

env_config = dict(id="two_segments_v2", random=1)
env = make_env(**env_config)
sample_train, sample_val, sample_test = make_data(env_config, val=True)