import tt2_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

def train(ts):
    env = gym.make("TT2", render_mode="rgb_array")
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, 
                              training=True, 
                              norm_obs=True, 
                              norm_reward=True, 
                              clip_obs=1000.0, 
                              clip_reward=1000.0, 
                              #gamma=0.99, 
                              #epsilon=1e-08, 
                              #norm_obs_keys=None
                              )
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold = 81)                                         
    eval_callback = EvalCallback(vec_env,
                             log_path="logs/", 
                             #best_model_save_path="logs/",
                             eval_freq=int(5e5),
                             n_eval_episodes= 1000,
                             deterministic=True, 
                             callback_on_new_best=callback_on_best)
    
    #policy_kwargs = dict(net_arch=[16,16])

    model = PPO("MlpPolicy", vec_env, verbose=0,
                #n_epochs = 16,
                #learning_rate=  0.0002,
                #n_steps= 8192,
                #policy_kwargs=policy_kwargs
                #ent_coef= 0.01,
                ) 

    model.learn(total_timesteps=ts,progress_bar=True, callback=eval_callback)
    model.save("logs/ppo_tt")
    vec_env.save("logs/vec_normalize.pkl")
    env.close()

def test(eps):

    test_env = gym.make("TT2", render_mode="human")
    test_env = Monitor(test_env)
    vec_test_env = DummyVecEnv([lambda: test_env])
    vec_test_env = VecNormalize.load("logs/vec_normalize.pkl", vec_test_env)

    #model = PPO.load("logs/best_model")
    model = PPO.load("logs/ppo_tt")
    
    mean_rewards, _ = evaluate_policy(model, vec_test_env, n_eval_episodes=eps, deterministic=True, return_episode_rewards=True)
    print("Rewards:", mean_rewards)

        
if __name__ == '__main__':
    train(5e7)
    test(20)