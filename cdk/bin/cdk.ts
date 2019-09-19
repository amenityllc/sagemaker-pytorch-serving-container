#!/usr/bin/env node
import 'source-map-support/register';
import cdk = require('@aws-cdk/core');
import { SageMakerStack } from '../lib/sage-maker-stack';

const app = new cdk.App();
new SageMakerStack(app, 'Sagemaker-bert');
