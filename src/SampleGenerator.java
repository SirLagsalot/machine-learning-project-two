import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SampleGenerator {

    public static List<Sample> generateSamples(int numSamples, int numInputs, int maxInputVal, int numOutputs) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(numInputs, maxInputVal, numOutputs);
            sample.outputs[0] = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }

    private static double computeRosenbrockOutput(double[] inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.length - 1; i++) {
            sum += Math.pow(1 - inputs[i], 2) + 100 * Math.pow(inputs[i + 1] - Math.pow(inputs[i], 2), 2);
        }

        return sum;
    }

    public static List<Sample> generateEasySamples(int numSamples, int numInputs) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(numInputs);
            for (int j = 0; j < numInputs; j++) {
                sample.inputs[j] = (double) random.nextInt(5);
            }
            sample.outputs[0] = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }
}
