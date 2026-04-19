### Conclusions

Although the day variable may contain some temporal information, its inclusion does not improve clustering quality and slightly degrades performance (see experiment 0. and 1. conclusions). Therefore, it is excluded from the final model to ensure more compact and well-separated clusters. Peak performance: Without day K = 8 [Experiment 1](full1.ipynb), with day K = 12 [Experiment 0](full.ipynb) and 15 components. 

Less clusters indicate: Cleaner structure, fewer and stronger clusters and better separation (higher silhouette), with "day" more clusters needed, slightly worse metrics, most likely capturing noise or minor variations.