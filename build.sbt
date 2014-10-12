import AssemblyKeys._ 

seq(assemblySettings: _*)

name := "SimpleANN"

version := "0.01"

organization := "com.korpisystems"

organizationHomepage := Some(new URL("http://korpisystems.com"))

startYear := Some(2014)

licenses := Seq("MIT" -> new URL("http://opensource.org/licenses/MIT"))

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.1.3" % "test"

scalaVersion := "2.10.3"

jarName in assembly := "SimpleANN.jar"
