import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.models.rl.enviornments.sb3_torch_wrapper import SB3TorchWrapper


class EvalAgentAcc:
    def __init__(self, model_with_rl, rl_agent, dataset, tokenizer, max_num_tks_sample=1024, exit_indices=[]):
        self.model = model_with_rl
        self.rl_agent = SB3TorchWrapper(rl_agent).to("cuda")
        self.exits = exit_indices
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_num_tks_sample = max_num_tks_sample

    def get_optimal_exit_point(self, label, states):
        exit_index = -1
        last_layer_token = states[-1]['prediction']
        first_last_match = -1
        for idx, state in enumerate(states):
            if state["prediction"] == label and exit_index == -1:
                exit_index = idx
            if state["prediction"] == last_layer_token and first_last_match == -1:
                first_last_match = idx
        return exit_index, first_last_match, last_layer_token

    def eval_action_acc(self, max_num_actions=500, max_tks_per_sample=15, ctx=0.35):
        acc_exit = []
        acc_exit_label = []
        acc_exit_wrong = []
        acc_exit_opt = []
        num_actions = 0

        for sample in self.dataset:
            print(num_actions)

            if num_actions >= max_num_actions:
                break
            code = sample['code']
            code_tokens = self.tokenizer.encode(code)
            code_tokens = torch.tensor(code_tokens).to('cuda')

            curr_ctx = int(ctx * len(code_tokens))

            for num_tks in range(0, max_tks_per_sample):
                if num_actions >= max_num_actions:
                    break
                curr_ctx += 1
                context_tokens = code_tokens[:curr_ctx]
                if len(context_tokens) == 0 or len(context_tokens) > self.max_num_tks_sample:
                    continue  # skip samples that are too short or too long

                label = code_tokens[curr_ctx]
                input_ids = context_tokens.clone().detach().unsqueeze(0).to('cuda')

                with torch.no_grad():
                    states = self.model(input_ids=input_ids)
                    optimal_exit_index, first_last_match, last_layer_token = self.get_optimal_exit_point(label, states)

                    for j, state in enumerate(states):
                        action_logits = self.rl_agent(state['hidden_state'][0, -1, :])
                        action_probabilities = F.softmax(action_logits, dim=0)
                        action = torch.argmax(action_probabilities).item()

                        if action == 1 and j in self.exits:
                            num_actions += 1

                            if state['prediction'] == label:
                                if j == optimal_exit_index:
                                    acc_exit_opt.append(1)

                                else:
                                    # still "good" exit, since correct
                                    acc_exit_label.append(1)

                                acc_exit.append(1)
                            elif optimal_exit_index == -1:
                                # no perfect exit exists
                                acc_exit_wrong.append(1)
                                acc_exit.append(1)
                            else:
                                # wrong exit
                                acc_exit.append(0)
                            break

        print(f"Exit Accuracy: {sum(acc_exit) / len(acc_exit)}")
        print(f"exit lengths: {len(acc_exit)}")

        return sum(acc_exit) / len(acc_exit), sum(acc_exit_wrong) / len(acc_exit), sum(acc_exit_label) / len(
            acc_exit), sum(acc_exit_opt) / len(acc_exit)

    def plot_action_acc(self, max_num_actions=500, max_tks_per_sample=15, ctx=0.35):
        acc_exit, exit_wrong, exit_label, exit_opt = self.eval_action_acc(max_num_actions, max_tks_per_sample, ctx)

        plt.figure(figsize=(8, 6))
        exit_values = [exit_wrong, exit_label, exit_opt]
        exit_labels = ['Unfeasible Exit.', 'Correct label (non-optimal)', 'Optimal']

        plt.bar(exit_labels, exit_values, color=['red', 'green', 'orange'])
        plt.title(f'RL Agent Accuracy on Exits: {acc_exit * 100}% total Accuracy')
        plt.ylim(0, 1)

        # Display the plot
        plt.tight_layout()
        plt.savefig("plot_agent_acc.png")


if __name__ == '__main__':
    pass
