import java.util.ArrayList;

public class FeedForwardNetwork extends NeuralNetwork {

    private int numHiddenLayers;
    private int numNeuronsPerHiddenLayer;
    private boolean momentum;

    private IActivationFunction activationFunction;

    private ArrayList<Neuron> inputLayer;
    private ArrayList<ArrayList<Neuron>> hiddenLayers;
    private ArrayList<Neuron> outputLayer;

    public FeedForwardNetwork(int inputs, int outputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                              boolean momentum, IActivationFunction activationFunction) {
        super(inputs, outputs);
        this.numHiddenLayers = numHiddenLayers;
        this.numNeuronsPerHiddenLayer = numNeuronsPerHiddenLayer;
        this.momentum = momentum;
        this.activationFunction = activationFunction;

        initializeNeurons();
    }

    @Override
    public void train() {

    }

    @Override
    public double approximate() {
        return 0;
    }

    private void initializeNeurons() {
        // Input Layer
        this.inputLayer = new ArrayList<>();
        for (int i = 0; i < this.inputs; i++) {
            inputLayer.add(new Neuron(this.activationFunction, this.numNeuronsPerHiddenLayer));
        }

        // Hidden layers
        this.hiddenLayers = new ArrayList<>();
        for (int i = 0; i < this.numHiddenLayers; i++) {
            // Build each layer
            ArrayList<Neuron> hiddenNeuronLayer = new ArrayList<>(this.numNeuronsPerHiddenLayer);
            if (i == this.numHiddenLayers - 1) {
                // Final hidden layer needs to have number of connections equal to the output layer size
                for (int j = 0; j < this.outputs; j++) {
                    hiddenNeuronLayer.add(new Neuron(this.activationFunction, this.outputs));
                }
            } else {
                for (int j = 0; j < this.numNeuronsPerHiddenLayer; j++) {
                    hiddenNeuronLayer.add(new Neuron(this.activationFunction, this.numNeuronsPerHiddenLayer));
                }
            }

            this.hiddenLayers.add(hiddenNeuronLayer);
        }

        // Output Layer
        this.outputLayer = new ArrayList<>();
        for (int i = 0; i < this.outputs; i++) {
            for (int j = 0; j < this.outputs; j++) {
                this.outputLayer.add(new Neuron(this.activationFunction, this.outputs));
            }
        }
    }
}
