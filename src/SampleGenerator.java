import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SampleGenerator {

    public static List<Sample> generateSamples(int numSamples, int maxInputVal, int maxInputs) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            int numInputs = random.nextInt(maxInputs);
            Sample sample = new Sample(numInputs, maxInputVal);
            sample.output = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }

    private static double computeRosenbrockOutput(double[] inputs) {
        int n = inputs.length;
        double sum = 0;

        for (int i = 0; i < n - 2; i++) {
            sum += Math.pow(1 - inputs[i], 2) + 100 * Math.pow(inputs[i + 1] - Math.pow(inputs[i], 2), 2);
        }

        return sum;
    }
}
