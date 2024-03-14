from keras.models import load_model
import sys
from cv2 import resize
import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found. Try to install with")
    print("pip install gymnasium[box2d]")
    sys.exit(1)


def preprocess_observation(obs):
    obs = resize(obs, (96, 96))
    obs = obs / 255.0
    return obs


def play(env, model):
    seed = 2000
    obs, _ = env.reset(seed=seed)

    # Drop initial frames
    action0 = 0
    for i in range(50):
        obs, _, _, _, _ = env.step(action0)

    done = False
    total_reward = 0  # Inizializza il totale delle ricompense
    while not done:
        processed_obs = preprocess_observation(obs)  # Preelaborazione dell'osservazione
        p = model.predict(np.array([processed_obs]))  # Adattamento per il modello
        action = np.argmax(p[0])  # Scelta dell'azione
        obs, reward, terminated, truncated, _ = env.step(
            action
        )  # Cattura la ricompensa
        total_reward += reward  # Aggiorna il totale delle ricompense
        print(
            f"Azione scelta: {action}, Ricompensa: {reward}, Totale Ricompensa: {total_reward}"
        )
        done = terminated or truncated

    return total_reward


env_arguments = {"domain_randomize": False, "continuous": False, "render_mode": "human"}

env_name = "CarRacing-v2"
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Carica il modello dopo aver eseguito il notebook
model_path = "firstmodel.keras"

model = load_model(model_path)
# simport visualkeras
# visualkeras.layered_view(model, to_file="output2.png")

play(env, model)
