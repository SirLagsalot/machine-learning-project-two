
import java.util.Arrays;
import java.util.List;

public class Tester {

    public static void main(String[] args) {
        int[] layers = new int[]{2, 200, 1};

        IFunctionApproximator neuralNetwork = new FeedForwardNetwork(2, 1, layers, 0.01, new SigmoidFunction());
        List<Sample> trainingSamples = SampleGenerator.generateSuperEasySamples(10000);
        neuralNetwork.train(trainingSamples);

        List<Sample> testSamples = SampleGenerator.generateSuperEasySamples(10);

        for (Sample sample : testSamples) {
            double networkOutput = neuralNetwork.approximate(sample.inputs)[0];
            System.out.println("Inputs: " + Arrays.toString(sample.inputs) + " Approx: " + networkOutput + " Actual: " + sample.outputs[0]);
            System.out.println("Error: " + Math.abs(networkOutput - sample.outputs[0]) + "\n");
        }
    }

    public static void crossValidate() {
        int numSamples = 10000;
        int inputs = 2;
        int maxInput = 5;
        int outputs = 1;
        int[] layers = {2, 50, 1};
        double learningRate = 0.01;
        double errorSum = 0;
        double[] totalErrorSum = new double[numSamples];

        List<Sample> samples = SampleGenerator.generateSamples(10000, 2, 5, 1);
        for (int k = 0; k < 5; k++) {
            List<Sample> train = samples.subList(0, (samples.size() / 2));
            List<Sample> test = samples.subList((samples.size() / 2), samples.size());

            IFunctionApproximator FFN = new FeedForwardNetwork(inputs, outputs, layers, learningRate, new SigmoidFunction());
            IFunctionApproximator RBN = new RadialBasisNetwork(inputs, outputs, 10);

            FFN.train(train);
            // radial.train(train);

            //iterates through test and calculates the aproximated values
            for (int i = 0; i < test.size(); i++) {
                Sample sample = test.get(i); //gets element in test at i
                double[] networkOutput = FFN.approximate(sample.inputs); //gets the approximated values of the inputs and puts them in a double array

                for (int j = 0; j < test.size(); j++) { //calculates the error for one input
                    double error = Math.abs(networkOutput[j] - sample.outputs[j]);
                    errorSum = errorSum + error; //adds the error from one input to the total error sum for the sample
                }
                totalErrorSum[i] = errorSum;
            }
            double mean = 0;
            double sum = 0;
            for (int i = 0; i < totalErrorSum.length; i++) {
                sum += totalErrorSum[i];
                mean = sum/totalErrorSum.length;
            }
            System.out.print("the mean of the Errors is: " + mean);

            double sd = 0;
            double standardDeviation = 0;
            for (int i = 0; i < totalErrorSum.length; i++) {
                sd += ((totalErrorSum[i] - mean)*(totalErrorSum[i] - mean)) / totalErrorSum.length - 1;
                standardDeviation = Math.sqrt(sd);
            }
            System.out.print("the standard deviation of the errors is: " + standardDeviation);
        }
    }
}
