package neuraalnetwork;

public class Connection {

    private Neuron connectionTo;
    private Weight weight;

    Connection(Neuron connectionTo, Weight weight) {
        this.connectionTo = connectionTo;
        this.weight = weight;
    }

    public Neuron getConnectionTo() {
        return connectionTo;
    }

    public double getWeight() {
        return weight.getWeight();
    }
}
