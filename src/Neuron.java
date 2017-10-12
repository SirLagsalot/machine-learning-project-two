import java.util.Random;

public class Neuron {

    private double output;
    private double bias;
    private double delta;
    private double[] weights;
    private double[] previousWeightUpdates;

    public Neuron(int prevLayerSize) {
        Random random = new Random();

        this.weights = new double[prevLayerSize];
        this.previousWeightUpdates = new double[prevLayerSize];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = random.nextDouble() / 100000000;
        }
        this.bias = random.nextDouble() / 10000000;
    }

    public double getWeight(int index) {
        return this.weights[index];
    }

    public void setWeight(int index, double weight) {
        this.weights[index] = weight;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getPreviousWeightUpdate(int index) {
        return previousWeightUpdates[index];
    }

    public void setPreviousWeightUpdate(int index, double previousWeightUpdates) {
        this.previousWeightUpdates[index] = previousWeightUpdates;
    }
}