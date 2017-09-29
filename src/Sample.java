import java.util.Random;

public class Sample {
    double[] inputs;
    double output;

    public Sample() {

    }

    public Sample(int numInputs, int maxInputVal) {
        this.inputs = new double[numInputs];

        Random random = new Random();
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = random.nextDouble() % maxInputVal;
        }
    }
}
