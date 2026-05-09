from submission.facade_dqn import train, plot_metrics

# Step 1: learn the basics on small board
metrics = train(size=5, episodes=5_000)

# Step 2: medium board
metrics = train(size=7, episodes=15_000)

# Step 3: only then go to 11x11
metrics = train(size=11, episodes=50_000)

plot_metrics(metrics)