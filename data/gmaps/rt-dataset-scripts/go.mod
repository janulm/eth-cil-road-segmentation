module main

go 1.22.4

replace github.com/mitroadmaps/roadtracer/dataset/lib => /home/paul7/go/src/rt_lib

require (
	github.com/IronSublimate/gomapinfer v0.0.0-20221118133313-ba940be8733c
	github.com/mitroadmaps/roadtracer/dataset/lib v0.0.0-00010101000000-000000000000
)

require (
	github.com/ajstarks/svgo v0.0.0-20211024235047-1546f124cd8b // indirect
	github.com/dhconnelly/rtreego v1.2.0 // indirect
	github.com/qedus/osmpbf v1.2.0 // indirect
	google.golang.org/protobuf v1.26.0 // indirect
)
