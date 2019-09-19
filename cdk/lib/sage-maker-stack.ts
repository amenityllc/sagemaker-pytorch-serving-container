import cdk = require('@aws-cdk/core');
import sagemaker = require('@aws-cdk/aws-sagemaker');
import iam = require('@aws-cdk/aws-iam');
import s3 = require('@aws-cdk/aws-s3');
import {ManagedPolicy} from "@aws-cdk/aws-iam";

const containerImage = "438190081246.dkr.ecr.eu-central-1.amazonaws.com/sage:latest";
const bucketName = "sagemaker-eu-central-1-438190081246";
const modelUrl = `s3://${bucketName}/sage-2019-09-18-08-51-14-622/sourcedir.tar.gz`;
const environment = "dev";
const releaseName = environment + '-' + `bert-${new Date().getTime()}`;

export class SageMakerStack extends cdk.Stack {
    constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
        super(scope, id, props);

        // Create a role that sagemaker can use which can access the model S3 bucket
        const sagemakerRole = new iam.Role(this, 'SagemakerRole', {
            assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com')
        });
        let sageMakerPolicy = ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess");
        sagemakerRole.addManagedPolicy(sageMakerPolicy);
        const bucket = s3.Bucket.fromBucketName(this, 'ImportedBucket', bucketName);
        bucket.grantReadWrite(sagemakerRole);


        // Create the SageMaker model
        const model = new sagemaker.CfnModel(this, 'model', {
            primaryContainer: {
                image: containerImage,
                modelDataUrl: modelUrl
            },
            executionRoleArn: sagemakerRole.roleArn,
            modelName: releaseName
        });

        // Create the sagemaker endpoint config for the sagemaker model
        const endpointConfig = new sagemaker.CfnEndpointConfig(this, 'endpointconfig', {
            productionVariants: [
                {
                    initialInstanceCount: 1,
                    initialVariantWeight: 1,
                    modelName: releaseName,
                    variantName: "AllTraffic",
                    instanceType: "ml.t2.medium",

                }
            ],
            endpointConfigName: releaseName
        });

        endpointConfig.node.addDependency(model);

        let cfnEndpoint = new sagemaker.CfnEndpoint(this, 'sagemaker-endpoint',
            {
                endpointConfigName: endpointConfig.endpointConfigName || "undefined",
            });

        cfnEndpoint.node.addDependency(endpointConfig)

    }
}
