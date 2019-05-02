import numpy as np
import stl
import cv2 as cv
import sys

np.set_printoptions(threshold=sys.maxsize)

MAX_THICK = 6.0
MIN_THICK = 1

def main():
	rawImg = cv.imread('batman.jpg', flags=cv.IMREAD_COLOR)
	grayImg = cv.cvtColor(rawImg, cv.COLOR_BGR2GRAY)

	maxVal = np.max(grayImg)
	minVal = np.min(grayImg)
	valRange = maxVal - minVal

	thickRange = MAX_THICK - MIN_THICK

	pixelThicknesses = np.zeros_like(grayImg)

	pixelThicknesses = MAX_THICK - (thickRange * (grayImg - minVal) / valRange)

	pixelMidpoints = np.zeros((grayImg.shape[0] - 1, grayImg.shape[1] - 1), dtype=float)

	for rowNum, rowVals in enumerate(pixelMidpoints):
		for colNum, val in enumerate(rowVals):
			pixelMidpoints[rowNum, colNum] = np.mean([pixelThicknesses[rowNum, colNum], pixelThicknesses[rowNum, colNum + 1], pixelThicknesses[rowNum + 1, colNum], pixelThicknesses[rowNum + 1, colNum + 1]])

	numFrontFaces = 4 * (grayImg.shape[0] - 1) * (grayImg.shape[1] - 1)
	numBackFaces = 2 * (grayImg.shape[0] - 1) * (grayImg.shape[1] - 1)
	numSideFaces = 2 * (grayImg.shape[0] - 1)
	numTopBotFaces = 2 * (grayImg.shape[1] - 1)

	numFaces = numFrontFaces + numBackFaces + (numSideFaces * 2) + (numTopBotFaces * 2)

	cube = stl.mesh.Mesh(np.zeros(numFaces, dtype=stl.mesh.Mesh.dtype))

	faceCounter = 0

	# Front
	for rowNum, rowMidpoints in enumerate(pixelMidpoints):
		for colNum, midpoint in enumerate(rowMidpoints):
			verticies = np.zeros((4,3), dtype=float)
			verticies[0,:] = np.array([colNum, pixelThicknesses[rowNum, colNum], grayImg.shape[0] - rowNum])
			verticies[1,:] = np.array([colNum + 1, pixelThicknesses[rowNum, colNum + 1], grayImg.shape[0] - rowNum])
			verticies[2,:] = np.array([colNum + 1, pixelThicknesses[rowNum + 1, colNum + 1], grayImg.shape[0] - (rowNum + 1)])
			verticies[3,:] = np.array([colNum, pixelThicknesses[rowNum + 1, colNum], grayImg.shape[0] - (rowNum + 1)])
			verticieMid = [colNum + 0.5, midpoint, grayImg.shape[0] - (rowNum + 0.5)]

			for i in range(4):
				cube.vectors[faceCounter, 0] = verticies[i]
				cube.vectors[faceCounter, 1] = verticieMid
				cube.vectors[faceCounter, 2] = verticies[(i + 1) % 4]
				cube.normals[faceCounter] = [0,1,0]
				faceCounter += 1

	# Back
	for rowNum, rowMidpoints in enumerate(pixelMidpoints):
		for colNum, midpoint in enumerate(rowMidpoints):
			verticies = np.zeros((4,3), dtype=float)
			verticies[0,:] = np.array([colNum, 0, grayImg.shape[0] - rowNum])
			verticies[1,:] = np.array([colNum + 1, 0, grayImg.shape[0] - rowNum])
			verticies[2,:] = np.array([colNum + 1, 0, grayImg.shape[0] - (rowNum + 1)])
			verticies[3,:] = np.array([colNum, 0, grayImg.shape[0] - (rowNum + 1)])

			for i in range(2):
				cube.vectors[faceCounter, 0] = verticies[(i * 2)]
				cube.vectors[faceCounter, 1] = verticies[((i * 2) + 1) % 4]
				cube.vectors[faceCounter, 2] = verticies[((i * 2) + 2) % 4]
				cube.normals[faceCounter] = [0,-1,0]
				faceCounter += 1

	# Top
	for colNum, _ in enumerate(pixelThicknesses[0,0:-1]):
		verticies = np.zeros((4,3), dtype=float)
		verticies[0,:] = np.array([colNum, pixelThicknesses[0, colNum], grayImg.shape[0]])
		verticies[1,:] = np.array([colNum + 1, pixelThicknesses[0, colNum + 1], grayImg.shape[0]])
		verticies[2,:] = np.array([colNum + 1, 0, grayImg.shape[0]])
		verticies[3,:] = np.array([colNum, 0, grayImg.shape[0]])

		for i in range(2):
			cube.vectors[faceCounter, 0] = verticies[(i * 2)]
			cube.vectors[faceCounter, 1] = verticies[((i * 2) + 1) % 4]
			cube.vectors[faceCounter, 2] = verticies[((i * 2) + 2) % 4]
			cube.normals[faceCounter] = [0,0,1]
			faceCounter += 1

	# Bottom
	for colNum, _ in enumerate(pixelThicknesses[-1,0:-1]):
		verticies = np.zeros((4,3), dtype=float)
		verticies[0,:] = np.array([colNum, pixelThicknesses[-1, colNum], 1])
		verticies[1,:] = np.array([colNum + 1, pixelThicknesses[-1, colNum + 1], 1])
		verticies[2,:] = np.array([colNum + 1, 0, 1])
		verticies[3,:] = np.array([colNum, 0, 1])

		for i in range(2):
			cube.vectors[faceCounter, 0] = verticies[(i * 2)]
			cube.vectors[faceCounter, 1] = verticies[((i * 2) + 1) % 4]
			cube.vectors[faceCounter, 2] = verticies[((i * 2) + 2) % 4]
			cube.normals[faceCounter] = [0,0,-1]
			faceCounter += 1

	# Left
	for rowNum, _ in enumerate(pixelThicknesses[0:-1,0]):
		verticies = np.zeros((4,3), dtype=float)
		verticies[0,:] = np.array([0, pixelThicknesses[rowNum, 0], grayImg.shape[0] - rowNum])
		verticies[1,:] = np.array([0, pixelThicknesses[rowNum + 1, 0], grayImg.shape[0] - (rowNum + 1)])
		verticies[2,:] = np.array([0, 0, grayImg.shape[0] - (rowNum + 1)])
		verticies[3,:] = np.array([0, 0, grayImg.shape[0] - rowNum])

		for i in range(2):
			cube.vectors[faceCounter, 0] = verticies[(i * 2)]
			cube.vectors[faceCounter, 1] = verticies[((i * 2) + 1) % 4]
			cube.vectors[faceCounter, 2] = verticies[((i * 2) + 2) % 4]
			cube.normals[faceCounter] = [-1,0,0]
			faceCounter += 1

	# Right
	for rowNum, _ in enumerate(pixelThicknesses[0:-1,-1]):
		verticies = np.zeros((4,3), dtype=float)
		verticies[0,:] = np.array([grayImg.shape[1] - 1, pixelThicknesses[rowNum, -1], grayImg.shape[0] - rowNum])
		verticies[1,:] = np.array([grayImg.shape[1] - 1, pixelThicknesses[rowNum + 1, -1], grayImg.shape[0] - (rowNum + 1)])
		verticies[2,:] = np.array([grayImg.shape[1] - 1, 0, grayImg.shape[0] - (rowNum + 1)])
		verticies[3,:] = np.array([grayImg.shape[1] - 1, 0, grayImg.shape[0] - rowNum])

		for i in range(2):
			cube.vectors[faceCounter, 0] = verticies[(i * 2)]
			cube.vectors[faceCounter, 1] = verticies[((i * 2) + 1) % 4]
			cube.vectors[faceCounter, 2] = verticies[((i * 2) + 2) % 4]
			cube.normals[faceCounter] = [1,0,0]
			faceCounter += 1

	print(faceCounter, cube.vectors.shape[0])
	cube.save('probsWrong.stl')

if __name__ == '__main__':
	main()
