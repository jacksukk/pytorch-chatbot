
def main():
	with open('data/movie_subtitles.txt') as f:
		total = f.readlines()
	test = total[-50000:]
	train = total[:-50000]
	with open('data/train.txt', 'w') as f:
		for line in train:
			f.write(line)
			#f.write('\n')
	with open('data/test.txt', 'w') as f:
		for line in test:
			f.write(line)
			#f.write('\n')
if __name__ == '__main__':
	main()