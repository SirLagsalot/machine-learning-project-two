public interface INeuron {
    double[] getWeights();
    int compute(double[] inputs);
}
