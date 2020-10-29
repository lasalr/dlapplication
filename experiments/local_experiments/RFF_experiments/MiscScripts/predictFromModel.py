import pickle
import sys
from DLplatform.parameters.vectorParameters import VectorParameter
from DLplatform.learning.batch.sklearnClassifiers import LinearSVC

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")


if __name__ == "__main__":
    with open(
            '../../examples/averagingAtTheEnd/Results/Results-HPC/LinearSVC_Radon point_2020-10-12_22-56-24/coordinator/currentAveragedState', 'rb') as f:
        averaged_model = pickle.load(f)

    print(type(averaged_model))
    svc_learner = LinearSVC(regParam=0.001, dim=18)

    svc_learner.setModel(param=averaged_model, setReference=True)

    svc_learner.setParameters(param=averaged_model)

    svc_learner.model.predict()