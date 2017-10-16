import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {

    public final int size;

    private List<Double> weights;
    private double activation;
    private double delta;
    private double bias;

    private double[] inputs;
    private double[] outputs;
    private double[] mean;

    private IActivationFunction activationFunction;

    // Feed-forward network constructor
    public Neuron(int connections, IActivationFunction activationFunction) {
        this.size = connections;
        this.activationFunction = activationFunction;
        this.initializeWeights();
    }

    // Radial basis network constructor
    public Neuron(int connections) {
        this.size = connections;
    }

    // Set all weights to a random value between [-0.5, 0.5]
    private void initializeWeights() {
        Random random = new Random(System.nanoTime());
        this.weights = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.weights.add(random.nextDouble() - 0.5);
        }
        this.bias = random.nextDouble() - 0.5;
    }

    public double execute(double[] inputs, boolean shouldUseActivationFunction) {
        double outputSum = bias;
        for (int i = 0; i < size; i++) {
            outputSum += inputs[i] * weights.get(i);
        }
        this.activation = shouldUseActivationFunction ? this.activationFunction.compute(outputSum) : outputSum;
        return activation;
    }

    public double getWeight(int index) {
        return this.weights.get(index);
    }

    public double getOutput() {
        return this.activation;
    }

    public double getDelta() {
        return this.delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public void updateBias(double increment) {
        this.bias += increment;
    }

    public void updateWeight(int index, double increment) {
        this.weights.set(index, this.weights.get(index) - increment);
    }

    public double[] getMean() {
        return mean;
    }

    public void setMean(double[] mean) {
        this.mean = mean;
    }

    public double[] getInputs() {
        return inputs;
    }

    public double getInput(int index) {
        return this.inputs[index];
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public double[] getOutputs() {
        return outputs;
    }

    public void setOutputs(double[] outputs) {
        this.outputs = outputs;
    }
}