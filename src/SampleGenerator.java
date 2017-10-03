import java.util.ArrayList;
import java.util.List;

public class SampleGenerator {

    public static List<Sample> generateSamples(int numSamples, int numInputs, int maxInputVal) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(numInputs, maxInputVal);
            sample.output = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }

    private static double computeRosenbrockOutput(Double[] inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.length - 1; i++) {
            sum += Math.pow(1 - inputs[i], 2) + 100 * Math.pow(inputs[i + 1] - Math.pow(inputs[i], 2), 2);
        }

        return sum;
    }
}
