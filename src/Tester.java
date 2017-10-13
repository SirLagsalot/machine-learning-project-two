
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

public class Tester {

    /* Tunable Parameters */
    private static final int numInputs = 2;
    private static final int numOutputs = 1;
    private static final int maxInputVal = 3;
    private static final int numSamples = 10000;
    private static final int[] layers = new int[]{numInputs, 5, numOutputs};    // Size of each layer
    private static final int batchSize = 5;
    private static final double learningRate = 0.05;
    private static final double momentum = 0.4;

    public static void main(String[] args) {
        IFunctionApproximator neuralNetwork = new FeedForwardNetwork(layers, learningRate, batchSize, momentum, new SigmoidFunction());
        List<Sample> trainingSamples = SampleGenerator.generateSamples(numSamples, numInputs, maxInputVal, numOutputs);
        neuralNetwork.train(trainingSamples);

        List<Sample> testSamples = SampleGenerator.generateSamples(numSamples, numInputs, maxInputVal, numOutputs);

        double error = 0.0;
        for (Sample sample : testSamples) {
            double output = neuralNetwork.approximate(sample.inputs)[0];
            error += Math.abs(output - sample.outputs[0]);
        }

        System.out.println("Avg Error: " + error / numSamples);
    }

    // Execute a 5x2 cross validation for both networks computing the mean and standard deviation of their errors
    public static void crossValidate() {
        int numSamples = 10000;
        int inputs = 2;
        int maxInput = 5;
        int outputs = 1;
        int[] layers = {2, 50, 1};
        double learningRate = 0.01;
        List<Sample> samples = SampleGenerator.generateSamples(numSamples, inputs, maxInput, outputs);

        IFunctionApproximator FFN;
        IFunctionApproximator RBN;

        List<Double> ffnErrors = new ArrayList<>();
        List<Double> rbfErrors = new ArrayList<>();

        for (int k = 0; k < 5; k++) {
            Collections.shuffle(samples);
            List<Sample> set1 = samples.subList(0, (samples.size() / 2));
            List<Sample> set2 = samples.subList((samples.size() / 2), samples.size());

            FFN = new FeedForwardNetwork(layers, learningRate, batchSize, momentum, new SigmoidFunction());
            RBN = new RadialBasisNetwork(numInputs, numOutputs, 10);

            ffnErrors.addAll(computeFold(set1, set2, FFN));
            rbfErrors.addAll(computeFold(set1, set2, RBN));

            FFN = new FeedForwardNetwork(layers, learningRate, batchSize, momentum, new SigmoidFunction());
            RBN = new RadialBasisNetwork(numInputs, numOutputs, 10);

            ffnErrors.addAll(computeFold(set2, set1, FFN));
            rbfErrors.addAll(computeFold(set2, set1, RBN));
        }

        double mean = calcMean(ffnErrors);
        double SD = calcStandardDeviation(mean, ffnErrors);
        printStats(mean, SD, "Feed Forward");

        mean = calcMean(rbfErrors);
        SD = calcStandardDeviation(mean, rbfErrors);
        printStats(mean, SD, "Radial Basis");
    }

    private static List<Double> computeFold(List<Sample> trainSet, List<Sample> testSet, IFunctionApproximator network) {
        network.train(trainSet);
        return getApproximationErrors(testSet, network);
    }

    // Iterates through testing set and calculates the approximated values and the error of the samples of the supplied network
    private static List<Double> getApproximationErrors(List<Sample> testSet, IFunctionApproximator network) {
        List<Double> totalError = new ArrayList<>(testSet.size());
        for (Sample sample : testSet) {
            // Get the network's approximation
            double[] networkOutput = network.approximate(sample.inputs);

            // Add the error for each sample to the total error
            totalError.add(IntStream
                    .range(0, networkOutput.length)
                    .mapToDouble(j -> Math.abs(networkOutput[j] - sample.outputs[j]))
                    .sum());
        }
        return totalError;
    }


    // Calculates the mean of all the samples errors
    private static double calcMean(List<Double> totalError) {
        return totalError
                .stream()
                .reduce(0.0, Double::sum) / totalError.size();
    }

    // Calculates the standard deviation of the provided errors
    private static double calcStandardDeviation(double average, List<Double> totalError) {
        return Math.sqrt(totalError
                .stream()
                .mapToDouble(aDouble -> Math.pow((aDouble - average), 2) / totalError.size())
                .sum());
    }

    // Writes the mean ans standard deviation to std out
    private static void printStats(double mean, double standardDeviation, String networkType) {
        System.out.println("Network type: " + networkType);
        System.out.println("-------------------------------");
        System.out.println("Mean error: " + mean);
        System.out.println("Standard Dev: " + standardDeviation);
    }
}


