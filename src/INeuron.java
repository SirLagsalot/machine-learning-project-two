import java.util.ArrayList;

public interface INeuron {
    double[] getWeights();
    double propagate(ArrayList<Double> inputs);
}
