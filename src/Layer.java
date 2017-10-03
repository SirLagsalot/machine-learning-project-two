import java.util.ArrayList;

public class Layer {

    private int size;
    private ArrayList<Neuron> neurons;

    public Layer(int size, int inputs, IActivationFunction activationFunction) {
        this.size = size;
        this.neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.neurons.add(new Neuron(activationFunction, inputs));
        }
    }

    public ArrayList<Double> execute(ArrayList<Double> inputs) {
        ArrayList<Double> outputs = new ArrayList<>(this.neurons.size());
        for (int i = 0; i < size; i++) {
            outputs.add(this.neurons.get(i).propagate(inputs));
        }
        return outputs;
    }

    public int getNumNodes() {
        return this.size;
    }

    public int getNumInputs() {
        return this.neurons.get(0).getNumInputs();
    }
}
