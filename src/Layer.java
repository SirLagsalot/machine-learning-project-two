public class Layer {

    public Neuron[] neurons;
    public int size;

    public Layer(int size, int prevLayerSize) {
        this.size = size;
        this.neurons = new Neuron[size];

        for (int j = 0; j < this.size; j++) {
            this.neurons[j] = new Neuron(prevLayerSize);
        }
    }
}