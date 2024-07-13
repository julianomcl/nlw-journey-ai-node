# Use the AWS base image for Node 20
FROM public.ecr.aws/lambda/nodejs:20

# Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copy the package.json and package-lock.json files
COPY package*.json ${LAMBDA_TASK_ROOT}

# Install packages
RUN npm ci --only=production

# Copy the source code
COPY travelAgent.js ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["travelAgent.lambdaHandler"]
