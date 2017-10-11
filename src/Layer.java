import java.util.ArrayList;
import java.util.List;

public class Layer {

    public final int size;
    private ArrayList<Neuron> neurons;

    public Layer(int size, int prevLayerSize) {
        this.size = size;
        this.neurons = new ArrayList<>(size);

        for (int i = 0; i < this.size; i++) {
            this.neurons.add(new Neuron(prevLayerSize));
        }
    }

    public Neuron getNeuron(int index) {
        return this.neurons.get(index);
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }
}