import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class RadialBasisNetwork extends NeuralNetwork {

    private double learnRate;
    private int batchSize;
    private int epochs;
    private List<Neuron> hiddenLayer;
    private List<Neuron> outputLayer;
    private IActivationFunction activationFunction;

    private final int size;

    public RadialBasisNetwork(int inputs, int outputs, int hiddenNodes, double learnRate, int batchSize, int epochs) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.size = hiddenNodes;
        this.learnRate = learnRate;
        this.batchSize = batchSize;
        this.epochs = epochs;
        this.initializeNetwork();
    }

    @Override
    public void train(List<Sample> samples) {
        this.initializeMeans(samples);

        double maxDist = 0.0;
        for (int i = 0; i < this.size; i++) {
            for (int j = i + 1; j < this.size; j++) {
                double dist = this.computeDistance(this.hiddenLayer.get(i).getInputs(), this.hiddenLayer.get(j).getInputs());
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

    private void initializeNetwork() {
        // Set up hidden layer
        this.hiddenLayer = new ArrayList<>(this.size);
        for (int i = 0; i < size; i++) {
            this.hiddenLayer.add(new Neuron(this.numInputs));
        }
        // Set up output layer
        this.outputLayer = new ArrayList<>(this.numOutputs);
        for (int i = 0; i < this.numOutputs; i++) {
            this.outputLayer.add(new Neuron(this.size));
        }
    }


    public void initializeMeans(List<Sample> samples) {
        Random random = new Random(System.nanoTime());
        for (int i = 0; i < this.size; i++) {
            int initIndex = random.nextInt(samples.size());
            Sample targetSample = samples.get(initIndex);
            this.hiddenLayer.get(i).setMean(targetSample.inputs);
        }
    }

    // Compute the euclidean distance between the to arrays
    public double computeDistance(double[] x, double[] y) {
        assert x.length == y.length;

        double sum = IntStream
                .range(0, x.length)
                .mapToDouble(i -> Math.pow(x[i] - y[i], 2))
                .sum();

        return Math.sqrt(sum);
    }

    public void updateWeights(int index, double[] networkOutputs, double[] expectedOutputs) {
        //update the weights of each neuron
        assert networkOutputs.length == expectedOutputs.length;

        double[] partialErrors = this.computeOutputErrorGradient(networkOutputs, expectedOutputs);
        for (int i = 0; i < this.size; i++) {
            for (int j = 0; j < this.outputLayer.size(); j++) {
                Neuron outputNeuron = this.outputLayer.get(i);
                outputNeuron.updateWeight(i, partialErrors[j] * this.learnRate);
            }
        }
    }

    private double[] gaussian(double[] inputs) {
        double[] gauss = new double[hiddenLayer.size()];
        for (int i = 0; i < this.hiddenLayer.size(); i++) {
            gauss[i] = this.hiddenLayer.get(i).getWeight(0) * this.activationFunction.compute(computeDistance(inputs, this.hiddenLayer.get(i).getMean()));
        }
        return gauss;
    }

    private double sumOutputs(double[] gaussOutput) {
        return Arrays.stream(gaussOutput).sum();
    }
}
