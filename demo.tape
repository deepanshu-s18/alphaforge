Output demo.gif
Set FontSize 14
Set Width 1200
Set Height 700
Set TypingSpeed 35ms

Type@400ms "bash scripts/show_hypotheses.sh"
Enter
Sleep 4s

Type@600ms "alphaforge 'post-earnings drift in megacap tech' --out /tmp/demo"
Enter
Sleep 14s

Type@400ms "cat /tmp/demo/*/report.md | sed -n '1,30p'"
Enter
Sleep 5s

Type@400ms "python evals/harness.py --limit 1"
Enter
Sleep 10s
