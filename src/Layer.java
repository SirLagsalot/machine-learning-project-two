import java.util.ArrayList;

public class Layer {

    private int size;
    private ArrayList<Neuron> neurons;

    public Layer(int size, int connections, IActivationFunction activationFunction) {
        this.size = size;
        this.neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.neurons.add(new Neuron(activationFunction, connections));
        }
    }

    public double[] execute(double[] inputs) {
        double[] outputs = new double[size];
        for (int i = 0; i < size; i++) {
            outputs[i] = this.neurons.get(i).propagate(inputs);
        }
        return outputs;
    }
}
