#!/usr/bin/python3
# This software is licensed under the GNU General Public License v3.0
# This software comes with ABSOLUTELY NO WARRANTY.
# The author does not accept responsibility for anything!
# Proof-of-concept for analyzing a .wav file into a text file containing pitches and duration. Midi support may be implemented later
# @author michael-land
# Usage: python3 programName name_of_audio_file.wav
from sys import argv
from scipy.io import wavfile
import wave, math, numpy
from math import log2, pow

programName = "BachAgain"
A4 = 440 # frequency of A4
# reads a wavefile with the filename @param fileName
# @return a tuple containing sampling rate and list of amplitudes
def read_wav(fileName):
    wf = wave.open(fileName, 'rb')
    nChannels = wf.getnchannels()
    wf.close()
    fsamp, ampRaw = wavfile.read(fileName)
    if nChannels == 1:
        return (fsamp, ampRaw)
    else:
        # sums all channels for stereo audio
        ampList = numpy.sum(ampRaw, 1) # 0 indicats to sum along the column axis (preserves the rows)
        return (fsamp, numpy.transpose(ampList))

# takes in a list of amplitudes, then constructs a spectrogram
def make_spectrogram(ampList, fs, lsect):
    nRows = int(numpy.ceil(len(ampList) / lsect))
    nCols = lsect
    # every row stores the results of an N-point dft of length lsect
    specgramtbr = numpy.empty([nRows, nCols], dtype = float)
    t = 0
    while t < nRows - 1:
        tempSegment = numpy.fft.fft(ampList[lsect * t : lsect * (t+1)])
        specgramtbr[t] = numpy.absolute(tempSegment)
        t += 1
    tempSegment = numpy.fft.fft(ampList[lsect * t : len(ampList)], lsect)
    # fft() pads zeros to reach lsect, calculating a lsect-point dft
    specgramtbr[t] = numpy.absolute(tempSegment)
    return specgramtbr

# property of a DFT (Discrete fourier transform): compared to the DTFT (discrete time
# fourier transform), there are no negative frequency values in the returned
# array, so the negative frequency values are represented periodically:
# i.e. if the section length on which to perform the DFT is 512, then
# the 0th to 255th element in the N-point DFT is the positive frequency value,
# corresponding to angular frequency (0, pi) in the DTFT;
# the 256th to 511th element in the N-point DFT is the negative frequency value
# corresponding to angular frequency (-pi, 0) int the DTFT wrapped around to (pi, 2 pi)
# this can be done due to the periodic nature of the DFT (discrete in both time and freq).
# For these purposes, we will ignore negative frequency values.
# (They are symmetrical to the positive ones anyways)
def trim_spectrogram(sg1):
    nRows, nCols = sg1.shape
    newCols = int(numpy.ceil(nCols / 2))
    sg2 = sg1[:,0:newCols]
    print(f"Trimmed array from {sg1.shape} to {sg2.shape}")
    return sg2

# Finds the strongest frequencies in a trimmed spectrogram. Assumes max frequency to be pi.
# pi corresponds to half the sampling frequency, according to the shannon-nyquist theorem
def find_strongest_frequencies(sg, fs):
    nRows, nCols = sg.shape
    threshold = 0.01 * numpy.amax(sg) # set threshold at 20 dB below max volume

    sg[sg < threshold] = 0 # zeros all values below the threshold, allows us to ignore noise frequencies

    indexOfMaxInEveryRow = numpy.argmax(sg, axis=1)
    # since there is only one max, and only the first index is returned, this does not support chords yet
    discreteFrequencyMultiplier = fs / nCols
    maxFreqs = indexOfMaxInEveryRow * discreteFrequencyMultiplier
    return maxFreqs

# Fits notes to frequencies using piano keys
def fit_pianokeys(freqs):
    pianoKeys = numpy.rint(12 * numpy.log2(freqs/A4) + 49) # div by 0 becomes -inf
    return pianoKeys

def freq_to_scientific(freqs):
    C0 = A4*pow(2, -4.75)
    keyNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    keysFromC0 = numpy.rint(12 * numpy.log2(freqs/C0)) # div by 0 becomes -inf
    tbr = ""
    for kfC0 in keysFromC0:
        if -1 < kfC0 < 100:
            aKeyFromC0 = int(kfC0)
            octaveNum = aKeyFromC0 // 12
            kn = aKeyFromC0 % 12
            tbr += keyNames[kn] + str(octaveNum) + " "
        #else do nothing to tbr
    return tbr

# Main execution code:
mainFN = ""
if len(argv) == 1:
    mainFN = input("Name of the audio file")
elif len(argv) == 2:
    mainFN = argv[1]
else:
    mainFN = argv[1]
    print(f'Usage: python3 {programName} audio_file_name.wav')
    print("Using first argument as audio file name.")

(fs, amps) = read_wav(mainFN)
sectionLengthDFT = 8192
sgVals = make_spectrogram(amps, fs, sectionLengthDFT)

numpy.savetxt(f"{mainFN}_spectrogram_values.csv", sgVals, delimiter=",")
print("spectrogram values successfully saved as .csv file")
# numpy.savetxt("test.csv", sgVals[0], delimiter=",") # savesfirst frame for testing purposes

sgTrimmed = trim_spectrogram(sgVals)

strongestFreqs = find_strongest_frequencies(sgTrimmed, fs)

pianoKey = fit_pianokeys(strongestFreqs)
numpy.savetxt(f"{mainFN}_piano.csv", pianoKey, delimiter=",")

scientificKeys = freq_to_scientific(strongestFreqs)
print(scientificKeys)
keyFile = open(f"{mainFN}_scientific.txt", "w")
keyFile.write(scientificKeys)
keyFile.write("\n")
keyFile.close()
