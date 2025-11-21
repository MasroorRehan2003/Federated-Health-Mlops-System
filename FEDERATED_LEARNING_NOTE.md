# Important Note About Federated Learning with Tree Models

## Tree Models vs Linear Models in Federated Learning

### The Challenge

Tree-based models (Random Forest, Gradient Boosting) **cannot be directly averaged** like linear models (Logistic Regression) in federated learning because:
- Trees are discrete structures (splits, nodes, leaves)
- You can't average two different tree structures
- Feature importances can be averaged, but the trees themselves cannot

### Current Implementation

For tree-based models, we use a **local training approach**:
1. Each client trains its own tree model on local data
2. Feature importances are extracted and aggregated as "parameters"
3. The aggregated importances are shared, but models remain local
4. Each client re-trains on local data each round

**This is NOT true federated averaging** - it's more like:
- Federated learning of feature importances
- Local training of tree structures
- Benefit: Models learn from local patterns while sharing feature importance insights

### Recommendation

For **true federated averaging** (FedAvg), use **Logistic Regression**:

```bash
# Server
python src/federated_learning/fl_server.py --model-type logistic_regression --rounds 10

# Clients
python src/federated_learning/fl_client.py --node-name hospital_A --model-type logistic_regression
python src/federated_learning/fl_client.py --node-name hospital_B --model-type logistic_regression
python src/federated_learning/fl_client.py --node-name hospital_C --model-type logistic_regression
```

With Logistic Regression:
- Model parameters (coefficients) can be directly averaged
- True federated averaging (FedAvg) works as intended
- Faster convergence
- Smaller communication overhead

### Why Tree Models Still Work

Even though tree models can't be truly federated, the current implementation still provides value:
- **Privacy**: Data never leaves local nodes
- **Local Adaptation**: Models adapt to local patterns
- **Feature Importance Sharing**: Shared insights about which features matter
- **Framework**: Demonstrates federated learning infrastructure

For production federated learning with tree models, consider:
- Federated boosting methods
- Model ensemble averaging (not parameter averaging)
- Federated decision trees (specialized algorithms)

