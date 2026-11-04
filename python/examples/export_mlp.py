from leantorch_exporter.exporter import LayerDescriptor, spec_from_descriptors
from leantorch_exporter.schema import ActivationType

spec = spec_from_descriptors([
    LayerDescriptor("linear", "fc1", [[0.5, -1.0], [0.0, 0.75]], [0.25, -0.5]),
    LayerDescriptor("activation", "relu", activation=ActivationType.RELU),
    LayerDescriptor("linear", "fc2", [[2.0, 1.0]], None)], [1, 2], "mlp")
spec.write_json("../Examples/models/mlp.json")
print("wrote ../Examples/models/mlp.json")
