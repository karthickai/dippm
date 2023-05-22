import dippm
import torchvision

model = torchvision.models.vgg16(pretrained=True)
model.eval()

#current dippm supports only A100 GPU
out  = dippm.predict(model, batch=8, input="3,244,244", device="A100")
print("Predicted Memory {0} MB, Energy {1} J, Latency {2} ms, MIG {3}".format(*out))


