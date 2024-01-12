from gradio_client import Client

client = Client("https://arsalan8-lfc-player-classifier.hf.space/--replicas/1l76g/")
result = client.predict(
		"https://i2-prod.dailystar.co.uk/incoming/article30459312.ece/ALTERNATES/s810/1_Darwin-Nunez-Unveils-his-New-Shirt-Number.jpg"
        ,api_name="/predict"
)
print(result)