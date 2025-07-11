import argparse
import torch
from torch.optim import Adam
from grpo_slm.env import ReasoningEnv
from grpo_slm.model import ReasoningModel
from grpo_slm.grpo import ppo_loss, grpo_loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="arithmetic")
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ReasoningEnv(max_value=20)
    model = ReasoningModel(args.model_name, device=device)
    optimizer = Adam(model.model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        prompts, responses, rewards, old_logps = [], [], [], []
        for _ in range(args.batch_size):
            prompt = env.reset()
            response = model.generate([prompt], max_new_tokens=5)[0]
            logp = model.log_probs([prompt], [response])[0]
            _, reward, _, _ = env.step(int(response.strip()) if response.strip().isdigit() else -1)

            prompts.append(prompt)
            responses.append(response)
            rewards.append(reward)
            old_logps.append(logp.detach())

        rewards = torch.tensor(rewards, device=device)
        old_logps = torch.stack(old_logps)
        # simple advantage = reward - baseline(mean)
        adv = rewards - rewards.mean()

        new_logps = model.log_probs(prompts, responses)
        loss = grpo_loss(old_logps, new_logps, adv)  # or ppo_loss for ablation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}, reward_mean={rewards.mean():.4f}")

if __name__ == "__main__":
    main()
