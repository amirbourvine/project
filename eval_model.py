def eval(model, horizon: int):
    vec_env = model.get_env()
    obs = vec_env.reset()
    profit_sum = 0.0
    total = 0.0
    demand = 0.0
    for _ in range(horizon):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if reward[0] > 0:
            profit_sum += reward[0]
        
        # print(f"info: {info}")

        if "demand" in info[0]:
            total += info[0]["total"]
            demand += info[0]["demand"]

        if done:
            break
    
    val = 0
    if total>0:
        val = ((demand/total)*100)

    print(f"profit_sum: {profit_sum}, demand_per: {val}")

    return profit_sum, val