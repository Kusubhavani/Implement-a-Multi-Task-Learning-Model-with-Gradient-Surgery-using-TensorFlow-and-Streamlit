from models.model import MultiTaskModel

model = MultiTaskModel()
model.build(input_shape=(None, 10))

with open("results/model_architecture.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))