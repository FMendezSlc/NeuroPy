% Felipe Méndez, INB, UNAM, Qro. Mexico, 2021
% Based on BeepDemo from Dr Peter Scarfe's Psychtoolbox Demos
% Added arduino control for external trigger
% Requires MATLAB Support Package for Arduino Hardware

% Clear the workspace
clearvars;
close all;
sca;

a = arduino(); % Initialize arduino

while 1
%     prompt = 'Do you want more? Y/N: ';
% str = input(prompt,'s');
% if str == 'y'
%     %writeDigitalPin(a, 'D11', 1);
% else
%     break
% end
 
if readDigitalPin(a, 'D52')
% Initialize Sounddriver
InitializePsychSound(1);

% Number of channels and Frequency of the sound
nrchannels = 2;
freq = 48000; % Not sound freq but device sampling rate

% Real sound freq used in MaekBeep
fq = 3000;

% How many times to we wish to play the sound
repetitions = 2;

% Length of the beep
beepLengthSecs = 0.02;

% Length of the pause between beeps
beepPauseTime = 0.5;

% Start immediately (0 = immediately)
startCue = 0;

% Should we wait for the device to really start (1 = yes)
% INFO: See help PsychPortAudio
waitForDeviceStart = 1;

% Open Psych-Audio port, with the follow arguements
% (1) [] = default sound device
% (2) 1 = sound playback only
% (3) 1 = default level of latency
% (4) Requested frequency in samples per second
% (5) 2 = stereo putput
pahandle = PsychPortAudio('Open', [], 1, 1, freq, nrchannels);

% Set the volume to half for this demo
PsychPortAudio('Volume', pahandle, 1.0);

% Make a beep which we will play back to the user
myBeep = MakeBeep(fq, beepLengthSecs);

% Fill the audio playback buffer with the audio data, doubled for stereo
% presentation
PsychPortAudio('FillBuffer', pahandle, [myBeep; myBeep]);

% Loop, ever heard of that?

for BeepNum = [1:repetitions]
    

% Start audio playback
PsychPortAudio('Start', pahandle, repetitions, startCue, waitForDeviceStart);

% Wait for the beep to end. Here we use an improved timing method suggested
% by Mario.
% See: https://groups.yahoo.com/neo/groups/psychtoolbox/conversations/messages/20863
% For more details.
[actualStartTime, ~, ~, estStopTime] = PsychPortAudio('Stop', pahandle, 1, 1, repetitions);

% Compute new start time for follow-up beep, beepPauseTime after end of
% previous one
startCue = estStopTime + beepPauseTime;

end

% Start audio playback
%PsychPortAudio('Start', pahandle, repetitions, startCue, waitForDeviceStart);

% Wait for stop of playback
%PsychPortAudio('Stop', pahandle, 1, 1);

% Close the audio device
PsychPortAudio('Close', pahandle);

writeDigitalPin(a, 'D11', 0);
end

end