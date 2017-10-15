import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;


public class Layer {

    public final int size;
    private List<Neuron> neurons;

    public Layer(int size, int connections, IActivationFunction activationFunction) {
        this.size = size;
        this.initializeNeurons(connections, activationFunction);
    }

    private void initializeNeurons(int connections, IActivationFunction activationFunction) {
        this.neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.neurons.add(new Neuron(connections, activationFunction));
        }
    }

    public double[] execute(double[] inputs, boolean shouldUseActivation) {
        double[] outputs = new double[size];

        IntStream.range(0, size).forEach(i -> outputs[i] = neurons.get(i).execute(inputs, shouldUseActivation));

        return outputs;
    }

    public Neuron getNeuron(int index) {
        return this.neurons.get(index);
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }
}