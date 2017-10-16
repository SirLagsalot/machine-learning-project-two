import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class RadialBasisNetwork extends NeuralNetwork {

    private double learnRate;
    private int batchSize;
    private int epochs;
    private List<Neuron> hiddenLayer;
    private IActivationFunction activationFunction;

    private final int size;

    public RadialBasisNetwork(int inputs, int outputs, int size, double learnRate, int batchSize, int epochs) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.size = size;
        this.learnRate = learnRate;
        this.batchSize = batchSize;
        this.epochs = epochs;
        this.hiddenLayer = new ArrayList<>(size);
    }

    @Override
    public void train(List<Sample> samples) {
        this.initializeMeans(samples);

        System.out.println(this.hiddenLayer.get(0).getOutput() + " " + this.hiddenLayer.get(1).getOutput());

        double maxDist = distance(this.hiddenLayer.get(0).getOutput(), this.hiddenLayer.get(1).getOutput());

        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                double dist = distance(this.hiddenLayer.get(i).getOutput(), this.hiddenLayer.get(j).getOutput());
                if (dist > maxDist) {
                    maxDist = dist;
                }
            }
        }

        GaussianFunction.setSigma(maxDist, size);

        for (int i = 0; i < epochs; i++) {
            double epochError = 0.0;
            double[] networkOutputs = new double[samples.size()];
            int k = 0;
            for (Sample sample : samples) {
                double[] gaussOutputs = gaussian(sample.inputs);
                networkOutputs[k] = sumOutputs(gaussOutputs);
                if (k % batchSize == 0) {

                    updateWeights(k, sample.outputs, networkOutputs);
                }

                epochError += this.calculateTotalError(sample.outputs, networkOutputs);
                k++;
            }
            System.out.println("Epoch: " + i + "\t\tError: " + epochError / samples.size());
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        double[] outputs = gaussian(inputs);
        double[] approx = new double[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            approx[i] = sumOutputs(outputs);
        }

        return approx;
    }


    public void initializeMeans(List<Sample> samples) {
        Random random = new Random(System.nanoTime());
        for (int i = 0; i < this.size; i++) {
            int initIndex = random.nextInt(samples.size());
            Sample targetSample = samples.get(initIndex);
            this.hiddenLayer.add(new Neuron(this.numInputs, targetSample.inputs, targetSample.outputs));
        }
    }

    public double distance(double x, double y) {
        return Math.pow(Math.abs(x - y), 2);
    }

    public void updateWeights(int index, double[] inputs, double[] expectedOutputs) {
        //update the weights of each neuron
        assert inputs.length == expectedOutputs.length;

        for (int i = 0; i < this.hiddenLayer.size(); i++) {
            double error = expectedOutputs[i] - inputs[0];
            Neuron currentNeuron = this.hiddenLayer.get(i);
            if (error >= 0) {
                currentNeuron.updateWeight(0, this.learnRate * this.activationFunction.computeDerivative(distance(inputs[0], currentNeuron.getInput(0))));
            } else {
                currentNeuron.updateWeight(0, -(this.learnRate * this.activationFunction.computeDerivative(distance(inputs[0], currentNeuron.getInput(0)))));
            }
        }
    }

    private double[] gaussian(double[] inputs) {
        double[] gauss = new double[hiddenLayer.size()];
        for (int i = 0; i < this.hiddenLayer.size(); i++) {
            gauss[i] = this.hiddenLayer.get(i).getWeight(0) * this.activationFunction.compute(distance(inputs[0], this.hiddenLayer.get(i).getMean()));
        }
        return gauss;
    }

    private double sumOutputs(double[] gaussOutput) {
        return Arrays.stream(gaussOutput).sum();
    }
}
