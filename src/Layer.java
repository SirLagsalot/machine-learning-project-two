import java.util.Random;

public class Layer {

    private int size;
    private int inputs;
    private double outputs[];
    private double deltas[];
    private double weights[][];
    private double bias[];
    private IActivationFunction activationFunction;

    public Layer(int size, int inputs, IActivationFunction activationFunction) {
        this.size = size;
        this.inputs = inputs;
        this.bias = new double[size];
        this.deltas = new double[size];
        this.activationFunction = activationFunction;
        this.weights = new double[size][inputs];
        this.initialize();
    }

    public double[] execute(double[] inputs) {
        this.outputs = new double[this.size];

        for (int i = 0; i < this.size; i++) {                   // Foreach neuron
            double neuronOutput = this.bias[i];
            for (int j = 0; j < inputs.length; j++) {           // Foreach input
                neuronOutput += this.weights[i][j] * inputs[j];
            }
            this.outputs[i] = this.activationFunction.compute(neuronOutput);
        }

        return this.outputs;
    }

    public int size() {
        return this.size;
    }

    public double getOutput(int index) {
        return this.outputs[index];
    }

    public double getDelta(int index) {
        return this.deltas[index];
    }

    public void setDelta(int index, double delta) {
        this.deltas[index] = delta;
    }

    public double getWeight(int neuronIndex, int inputIndex) {
        return this.weights[neuronIndex][inputIndex];
    }

    private void initialize() {
        Random random = new Random();

        for (int i = 0; i < this.size; i++) {
            this.bias[i] = 1;
            for (int j = 0; j < this.inputs; j++) {
                this.weights[i][j] = random.nextDouble();
            }
        }
    }
}
