$outfile = "int_benchmark.csv"
"File,Execution time,Solution" | Out-File -FilePath $outfile
Get-ChildItem "Instances" -Filter FLP*.txt | Foreach-Object {
    $result = python flp.py $_ --int
    $time = $result | Select-String -Pattern "Resolution time"
    $time = $time -replace "[a-zA-Z\s:+]", ""
    $solution = $result | Select-String -Pattern "Final solution"
    $solution = $solution -replace "[a-zA-Z\s:+]", ""
    "$_,$time,$solution" | Out-File -FilePath $outfile -Append
}