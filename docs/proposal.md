# Dual Path Qwen: Using General Purpose LLMs in High-Level Control and Reactive Settings

*Original proposal, verbatim (submitted October 2026). The current plan, which narrows scope, is in `docs/project-overview.md`.*

Jeet Parikh (jnp32), Farhan Baig (fb388), Kashvi Pundir (kp794), Hari Viswanathan (hv99)

## General Motivation

Recent multimodal language models like GPT-6 Astra and Claude Fable 5 have proven to be effective at high-level control tasks in spatially dependent environments. Much of this can likely be credited to increased abilities in computer use settings, where understanding of state action relationships is incredibly paramount. When placed into VLAs, the models do generalize well to robotic manipulation tasks. However, due to the size of the models, it is unreasonable to expect that we can use such systems for real-time robotics.

The motivating idea behind this project is to develop a system for using general-purpose LLMs in environments where we need both system 1 (fast) and system 2 (slow) thinking. The idea is that we can segment an LLM to act as both a fast-twitch action selector and a higher-level orchestrator. Let us consider a concrete example of such a scenario. Imagine a user asks for a robot to clean the dishes. The high-level planner module can break the task into a series of steps: move to the sink, rinse each dish, place dishes into the dishwasher, add detergent, and finally start the dishwasher. Then, for each of these objectives, the fast-twitch module will pick actions rapidly within a constrained action space for the robot (i.e. rotate arm x degrees, move forward x units, etc) in order to complete the subtask. Periodically, the high-level planner will be polled to check progress and redirect if needed.

Recent models like Jev target this system 1 level thinking by acting as classifiers over constrained action spaces. They are exceptionally cheap and fast, and thus well-suited to act as fast-twitch modules. Open source versions of Jev, like Kev, have proven that such abilities can be extracted from existing LLMs like Qwen.

The question we seek to answer in our project is whether such a model can be extracted from general-purpose models like Qwen and function alongside the base model to provide fast-twitch control. We propose a methodology to split general-purpose models like Qwen into both system 1 and 2 thinkers. We hypothesize that the early learned layers of such models are, by themselves, capable enough of containing the information needed to act as efficient classifiers, while using the entire depth of the model can provide accurate high-level reasoning.

## Related Work

Jev is a new model released by Typesafe AI that returns structured choices and probabilities from the LLM rather than raw text, so software can use its answers without interpreting generated text. We want to see whether we can build a similar decision-making interface inside Qwen, using a small classifier attached to an intermediate layer. This would be inspired by Jev's behavior, rather than copying its architecture.

The idea of getting predictions before the final layer is not new. DeeBERT and LayerSkip show how models can make earlier predictions, while UniFS explores shared fast and slow computation for robotic control. These provide relevant precedents, but do not establish how well a small head on frozen Qwen representations would perform as an action controller

What we want to investigate is whether the same input can give us two useful outputs at different depths: an immediate action after the first few layers, and a higher-level output after the remaining layers. We will test how reliable those early actions are, whether training them affects the final output, and how much faster the model becomes when we stop early.

## Execution Plan

We plan on starting with a pretrained Qwen model (8B parameters although we might test across different model sizes) and add support for two computational modes/paths. The first will be a slow/planning mode where the model uses its full depth to provide longer-horizon reasoning and task decomposition. This is the less interesting of the two paths since, effectively, we will just use the model as is in order to retain its abilities.

The second path will be the fast mode, where we will only utilize the first k layers (8-12 layers). After these layers, we will take the hidden state and use it as the input to a Jev-like classifier head. In order to train this head, we will freeze the first k layers and train it on the hidden state to assign probabilities to a set of actions within the constrained action space.

The final piece is the router that decides when a request requires the slow path versus the fast path. This will be done via two main candidate methods. The first will be a learned lightweight router that sits before the first layer of the model. The second will be a mechanism that triggers full depth reasoning if the classifier probabilities are uncertain (i.e. there is no clear winner in action suggestion). We plan to compare the results from both of these approaches to see which is more effective in practice. We also hope, as a stretch goal, to evaluate efficient inference techniques for our dual path architecture.

## Evaluation Plan

We plan on evaluating our dual path model on robotic manipulation benchmarks like LIBERO and Meta-World. Similar to other papers in the space, we will measure effectiveness via standard metrics like inference latency, task success rate, and used FLOPs. Some additional metrics we will track for the sake of understanding dual path effectiveness are the ratio between slow and fast path usage and fast path action probability distributions (to measure confidence in certain domains).

The comparison baselines we plan on using are existing exit early inference mechanisms, as well as similar split models that do not combine the fast and slow paths into a single model. Even within our dual-path architecture, we plan to vary fast path depth and routing mechanisms. We hope to see our architecture maintain similar task performance while using far less compute.

## AI Usage Statement

We plan on using AI tools for the purpose of literature review and code implementation. We do not expect to rely entirely on these models for coding assistance. However, it is likely that we will use them for debugging purposes and to write scripts for running things on the Yale cluster.

## References

- Elhoushi, M., et al. (2024). LayerSkip: Enabling early exit inference and self-speculative decoding. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*, 12622–12642.
- Palmer, J. (2026). Kev [Computer software]. GitHub. https://github.com/jaredpalmer/kev
- Sun, L., Guan, Z., Wang, C., Chen, Z., Yu, J., Li, Z., He, B., Sun, T., Cao, J., & Liu, L. (2026). UniFS: Unified fast-to-slow hierarchical architecture for vision-language-action models. arXiv. https://arxiv.org/abs/2606.22794
- TypeSafe AI. (n.d.). Introduction. TypeSafe AI Documentation. Retrieved October 5, 2026, from https://docs.typesafe.ai/introduction
- Xin, J., Tang, R., Lee, J., Yu, Y., & Lin, J. (2020). DeeBERT: Dynamic early exiting for accelerating BERT inference. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 2246–2251. https://aclanthology.org/2020.acl-main.204/
- Zhang, W., Wang, K., Ouyang, Y., Huang, X., Li, L., Su, K., Jin, W., Chai, W., Liang, H., Dou, Z., Chen, Y., & Chen, T. (2026). An unexpected robot policy: Early evaluations of GPT-6 Astra on RoboDojo and beyond. arXiv. https://doi.org/10.48550/arXiv.2609.24170
