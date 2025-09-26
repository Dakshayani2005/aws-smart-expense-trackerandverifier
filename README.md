🏗️ Architecture

This project follows a simple serverless architecture using AWS services to process and verify uploaded bills.

🔄 Workflow:

1.User uploads a bill to an S3 bucket (uploads).

2.S3 triggers a Lambda function (lambda_function.py).

3.Lambda runs a deep learning model (predict.py) to classify the bill as "real" or "fake".

4.If the bill is fake, it sends an alert via Amazon SNS.

5.If the bill is real, it uses Amazon Textract to extract text from the bill.

6.The extracted text is saved to another S3 bucket (outputs) and optionally processed further.

🧩 Components Involved:

1.Amazon S3 – to store input bills and output text files.

2.AWS Lambda – serverless compute to process files and run the model.

3.Amazon Textract – to extract raw text from scanned documents.

4.Amazon SNS – to send alerts for fake bills.

5.Deep Learning Model – a custom model to verify authenticity of bills.