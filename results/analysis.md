### Gradient Conflict Analysis

The cosine similarity values between task gradients fluctuate around zero,
with several noticeable negative spikes throughout training. These negative
values indicate the presence of gradient conflicts, where the optimization
directions of different tasks oppose each other.

In the early stages of training, conflicts are more frequent and pronounced,
suggesting that the model struggles to balance both tasks initially. As training
progresses, the fluctuations become more stable, although occasional conflicts
still occur. This behavior confirms that the tasks are not fully aligned and
justifies the use of PCGrad to mitigate gradient interference.

The visualization with highlighted conflict points (negative cosine similarity)
provides a clear indication of when and how often these conflicts arise.


### Shared Representation Analysis

Although explicit UMAP visualizations were not implemented in this version,
the behavior of the model provides indirect insights into the shared representation.

The stable convergence of both tasks suggests that the shared backbone is able
to learn a meaningful representation that supports both classification and
regression objectives. The reduction in training loss over time indicates that
the model successfully captures useful features for both tasks.

In a complete implementation, UMAP or t-SNE could be used to project the learned
feature embeddings into a 2D space. Ideally, well-separated clusters for different
task labels would indicate a strong shared representation.


### Final Performance Comparison

The final performance metrics, as recorded in `final_metrics.json`, compare the
baseline model with the PCGrad-enhanced model.

Both models demonstrate reasonable performance; however, the PCGrad model shows
more stable optimization behavior in the presence of gradient conflicts. By
resolving conflicting gradients, PCGrad allows the model to make more consistent
updates, leading to improved or more reliable performance across tasks.

Overall, the results indicate that PCGrad is effective in handling multi-task
learning scenarios where task objectives may interfere with each other.
