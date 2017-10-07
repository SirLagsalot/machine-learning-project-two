import java.util.ArrayList;

public class Layer {

    private int size;
    private ArrayList<Neuron> neurons;

    public Layer(int size, int inputs, double learningRate, IActivationFunction activationFunction) {
        this.size = size;
        this.neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.neurons.add(new Neuron(inputs, learningRate, activationFunction));
        }
    }

    public ArrayList<Double> execute(ArrayList<Double> inputs) {
        ArrayList<Double> outputs = new ArrayList<>(this.neurons.size());
        for (int i = 0; i < size; i++) {
            outputs.add(this.neurons.get(i).propagate(inputs));
        }
        return outputs;
    }

    public ArrayList<Neuron> getNeurons() {
        return this.neurons;
    }

    public int getSize() {
        return this.size;
    }
}
