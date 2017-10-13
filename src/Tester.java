
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    //execute 5X2 cross validation on both data sets
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
        printStats(mean, SD, "FFN");

        mean = calcMean(rbfErrors);
        SD = calcStandardDeviation(mean, rbfErrors);
        printStats(mean, SD, "RBF");
    }

    private static List<Double> computeFold(List<Sample> trainSet, List<Sample> testSet, IFunctionApproximator network) {
        network.train(trainSet);
        return getApproximationErrors(testSet, network);
    }

    //iterates through testing set and calculates the approximated values and the error of the samples of the feedforward network
    public static List<Double> getApproximationErrors(List<Sample> testSet, IFunctionApproximator network) {
        List<Double> totalError = new ArrayList<>(testSet.size());
        for (Sample sample : testSet) {
            double[] networkOutput = network.approximate(sample.inputs); //gets approximated input values
            double sampleError = 0;
            for (int j = 0; j < networkOutput.length; j++) {
                sampleError += Math.abs(networkOutput[j] - sample.outputs[j]);  //calculates sum of errors for the sample
            }
            //put the errorSum of each sample into an array of doubles (FFNtotalError used in mean and SD)
            totalError.add(sampleError);
        }
        return totalError;
    }


    //calculates the mean of all the samples errors
    public static double calcMean(List<Double> totalError) {
        final double[] sum = {0};

        totalError.forEach(error -> sum[0] += error);

        return sum[0] / totalError.size();
    }

    //calculates the standard deviation of all the samples errors
    public static double calcStandardDeviation(double average, List<Double> totalError) {
        double standardDeviation = 0;
        double sd = 0;

        for (int i = 0; i < totalError.size(); i++) {
            sd += Math.pow((totalError.get(i) - average), 2) / totalError.size();
            standardDeviation = Math.sqrt(sd);
        }
        return standardDeviation;
    }

    public static void printStats(double mean, double standardDeviation, String networkType) {
        System.out.println("The Mean of the " + networkType + " errors is: " + mean);
        System.out.println("The SD of the " + networkType + " errors is: " + standardDeviation);
    }
}


