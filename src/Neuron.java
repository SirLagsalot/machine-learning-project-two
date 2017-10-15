import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {

    public final int size;

    private List<Double> weights;
    private double activation;
    private double delta;
    private double bias;

    private IActivationFunction activationFunction;

    public Neuron(int connections, IActivationFunction activationFunction) {
        this.size = connections;
        this.activationFunction = activationFunction;
        this.initializeWeights();
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
    public void setOutput(double value){
        this.activation = value;
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
        this.weights.set(index, this.weights.get(index) + increment);
    }
}